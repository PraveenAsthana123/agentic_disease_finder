#!/usr/bin/env python3
"""Clinical Psychologist module — depression/anxiety assessment, coping & resilience,
seizure-emotion correlation, and CBT/ACT therapy planning for epilepsy patients.

ALL data is REAL (no synthetic) — drawn from the `assessments` table:
  PHQ-9 (26)  depression screen, item9 = passive/active suicidal ideation
  GAD-7 (25)  generalized anxiety screen
  NDDI-E (3)  Neurological Disorders Depression Inventory for Epilepsy
  QOLIE-31 (23) quality-of-life dimensions (coping/wellbeing proxy)
plus `seizure_diary` (25) for seizure-emotion correlation.

Mirrors the established expert-module pattern (ot_module / pharmacist_module).
"""

import json
import os
import sqlite3
from collections import Counter, defaultdict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinical.db")

# PHQ-9 severity bands (standard cutoffs)
PHQ9_BANDS = [(0, 4, "Minimal", "none/minimal"), (5, 9, "Mild", "watchful waiting"),
              (10, 14, "Moderate", "treatment plan (counseling/pharmacotherapy)"),
              (15, 19, "Moderately Severe", "active treatment"),
              (20, 27, "Severe", "immediate pharmacotherapy + specialist referral")]
# GAD-7 severity bands
GAD7_BANDS = [(0, 4, "Minimal", "none"), (5, 9, "Mild", "monitor"),
              (10, 14, "Moderate", "probable anxiety — assess/treat"),
              (15, 21, "Severe", "active treatment")]


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _rows(cur):
    return [dict(r) for r in cur.fetchall()]


def _band(score, bands):
    for lo, hi, label, action in bands:
        if lo <= score <= hi:
            return {"level": label, "action": action}
    return {"level": "out of range", "action": ""}


def _answers(row):
    try:
        return json.loads(row.get("answers_json") or "{}")
    except (ValueError, TypeError):
        return {}


# ─── 1. Depression / Anxiety Assessment ──────────────────────────────────

def depression_anxiety(patient_id=None):
    """PHQ-9 + GAD-7 + NDDI-E auto-scoring with severity bands and a
    suicide-risk escalation flag (PHQ-9 item 9 ≥ 1 → C-SSRS recommended)."""
    c = _conn()
    where = "AND patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()
    out = {"available": True, "instruments": {}, "patients": [], "alerts": []}

    by_patient = defaultdict(dict)
    for ins in ("PHQ9", "GAD7", "NDDIE"):
        rows = _rows(c.execute(
            f"SELECT patient_id, score, max_score, level, interpretation, answers_json, created_at "
            f"FROM assessments WHERE instrument = ? {where} ORDER BY created_at", (ins, *params)))
        bands = PHQ9_BANDS if ins == "PHQ9" else (GAD7_BANDS if ins == "GAD7" else None)
        scored = []
        for r in rows:
            sev = _band(r["score"], bands) if bands else {"level": r["level"] or "n/a", "action": ""}
            entry = {"patient_id": r["patient_id"], "score": r["score"], "max_score": r["max_score"],
                     "severity": sev["level"], "action": sev["action"]}
            # PHQ-9 item 9 — suicidal ideation
            if ins == "PHQ9":
                item9 = _answers(r).get("item9", 0) or 0
                entry["suicidal_ideation"] = int(item9)
                if item9 >= 1:
                    out["alerts"].append({
                        "patient_id": r["patient_id"], "type": "suicidal_ideation",
                        "severity": "HIGH" if item9 >= 2 else "MODERATE",
                        "detail": f"PHQ-9 item 9 = {item9} — escalate to C-SSRS + same-day safety assessment",
                    })
            scored.append(entry)
            by_patient[r["patient_id"]][ins] = entry
        out["instruments"][ins] = {
            "n": len(scored),
            "severity_distribution": dict(Counter(e["severity"] for e in scored)),
            "results": scored,
        }

    # per-patient combined depression+anxiety profile
    for pid, ins_map in sorted(by_patient.items()):
        phq = ins_map.get("PHQ9", {})
        gad = ins_map.get("GAD7", {})
        out["patients"].append({
            "patient_id": pid,
            "depression": phq.get("severity", "—"), "phq9_score": phq.get("score"),
            "anxiety": gad.get("severity", "—"), "gad7_score": gad.get("score"),
            "comorbid_anx_dep": bool(phq.get("score", 0) >= 10 and gad.get("score", 0) >= 10),
            "suicidal_ideation": phq.get("suicidal_ideation", 0),
        })
    c.close()
    out["summary"] = {
        "patients_assessed": len(out["patients"]),
        "suicide_risk_flags": len(out["alerts"]),
        "moderate_or_worse_depression": sum(1 for p in out["patients"] if (p["phq9_score"] or 0) >= 10),
        "moderate_or_worse_anxiety": sum(1 for p in out["patients"] if (p["gad7_score"] or 0) >= 10),
    }
    out["note"] = ("PHQ-9 item 9 screens passive/active suicidal ideation; any positive endorsement "
                   "triggers a C-SSRS escalation flag. AED side-effects can mimic depression — interpret "
                   "alongside medication review.")
    return out


# ─── 2. Coping & Resilience (QOLIE-31 dimensions as proxy) ────────────────

def coping_resilience(patient_id=None):
    """QOLIE-31 emotional/social wellbeing as a coping/resilience proxy
    (no Brief-COPE/AAQ-II data yet — uses live QoL dimensions honestly)."""
    c = _conn()
    where = "AND patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()
    rows = _rows(c.execute(
        f"SELECT patient_id, score, max_score, level, interpretation, answers_json, created_at "
        f"FROM assessments WHERE instrument = 'QOLIE31' {where} ORDER BY created_at", params))
    c.close()
    if not rows:
        return {"available": False, "error": "no QOLIE-31 records"}
    profiles = []
    for r in rows:
        score = r["score"] or 0
        resilience = ("High" if score >= 70 else "Moderate" if score >= 50 else "Low")
        profiles.append({
            "patient_id": r["patient_id"], "qolie31_score": score, "max": r["max_score"],
            "qol_level": r["level"] or r["interpretation"], "resilience_proxy": resilience,
            "therapy_target": ("maintenance" if resilience == "High"
                               else "coping-skills building" if resilience == "Moderate"
                               else "intensive resilience + CBT"),
        })
    dist = Counter(p["resilience_proxy"] for p in profiles)
    return {
        "available": True, "n": len(profiles), "profiles": profiles,
        "resilience_distribution": dict(dist),
        "note": ("QOLIE-31 used as a wellbeing/resilience proxy — Brief-COPE / AAQ-II not yet "
                 "in the dataset. Low resilience (<50) flags patients for coping-skills therapy."),
    }


# ─── 3. Seizure–Emotion Correlation ──────────────────────────────────────

def seizure_emotion_correlation(patient_id=None):
    """Relate seizure burden (seizure_diary count) to mood severity (PHQ-9/GAD-7)
    per patient — surfaces whether higher seizure frequency tracks worse mood."""
    c = _conn()
    where_s = "WHERE patient_id = ?" if patient_id else ""
    where_a = "AND patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()
    sz = _rows(c.execute(f"SELECT patient_id, COUNT(*) AS n FROM seizure_diary {where_s} GROUP BY patient_id", params))
    sz_count = {r["patient_id"]: r["n"] for r in sz}
    mood = {}
    for ins in ("PHQ9", "GAD7"):
        for r in _rows(c.execute(
                f"SELECT patient_id, score FROM assessments WHERE instrument = ? {where_a}", (ins, *params))):
            mood.setdefault(r["patient_id"], {})[ins] = r["score"]
    c.close()

    pairs = []
    for pid in sorted(set(sz_count) | set(mood)):
        n = sz_count.get(pid, 0)
        m = mood.get(pid, {})
        pairs.append({"patient_id": pid, "seizure_count": n,
                      "phq9": m.get("PHQ9"), "gad7": m.get("GAD7")})
    # simple correlation (seizure_count vs phq9) over patients with both
    both = [(p["seizure_count"], p["phq9"]) for p in pairs if p["phq9"] is not None]
    corr = None
    if len(both) >= 3:
        import statistics
        xs, ys = zip(*both)
        try:
            corr = round(statistics.correlation(xs, ys), 3)
        except (statistics.StatisticsError, ValueError):
            corr = None
    return {
        "available": True, "n_patients": len(pairs), "pairs": pairs,
        "seizure_vs_depression_corr": corr,
        "interpretation": ("Positive correlation suggests seizure burden tracks low mood — "
                           "prioritize integrated seizure-control + mood treatment."
                           if (corr or 0) > 0.3 else
                           "No strong linear seizure-mood coupling in this cohort; treat independently."),
        "note": "Pearson correlation over patients with both seizure-diary and PHQ-9 records (real data).",
    }


# ─── 4. Therapy Planning (CBT / ACT) ─────────────────────────────────────

def therapy_planning(patient_id=None):
    """Rule-based CBT/ACT therapy recommendations derived from live PHQ-9/GAD-7
    severity (transparent, score-driven — not a black box)."""
    da = depression_anxiety(patient_id)
    plans = []
    for p in da.get("patients", []):
        dep = p["phq9_score"] or 0
        anx = p["gad7_score"] or 0
        modalities, priority = [], "routine"
        if p.get("suicidal_ideation", 0) >= 1:
            modalities.append("Safety planning + crisis support (C-SSRS)")
            priority = "urgent"
        if dep >= 15:
            modalities.append("CBT for depression (behavioral activation + cognitive restructuring)")
            priority = "urgent" if priority != "urgent" else priority
        elif dep >= 10:
            modalities.append("CBT (cognitive restructuring) ± pharmacotherapy")
            priority = "high" if priority == "routine" else priority
        elif dep >= 5:
            modalities.append("Behavioral activation / psychoeducation")
        if anx >= 10:
            modalities.append("CBT for anxiety (exposure + relaxation) / ACT defusion")
            priority = "high" if priority == "routine" else priority
        elif anx >= 5:
            modalities.append("Relaxation training + worry management")
        if dep >= 10 and anx >= 10:
            modalities.append("ACT — values-based action for comorbid anx/dep + epilepsy adjustment")
        if not modalities:
            modalities.append("Maintenance / psychoeducation on epilepsy adjustment")
        plans.append({
            "patient_id": p["patient_id"], "depression": p["depression"], "anxiety": p["anxiety"],
            "priority": priority, "recommended_modalities": modalities,
            "sessions_suggested": 12 if priority in ("urgent", "high") else 6,
        })
    plans.sort(key=lambda x: {"urgent": 0, "high": 1, "routine": 2}[x["priority"]])
    return {
        "available": True, "n": len(plans), "plans": plans,
        "priority_distribution": dict(Counter(p["priority"] for p in plans)),
        "note": ("Transparent rule-based CBT/ACT targeting from live PHQ-9/GAD-7 scores. "
                 "Urgent = suicidal ideation or severe depression; clinician confirms before booking."),
    }


def full_dashboard(patient_id=None):
    """Combined Clinical Psychologist dashboard — all 4 modules."""
    return {
        "role": "Clinical Psychologist",
        "description": "Depression/anxiety assessment, coping & resilience, seizure-emotion "
                       "correlation, and CBT/ACT therapy planning for epilepsy patients",
        "modules": {
            "depression_anxiety": depression_anxiety(patient_id),
            "coping_resilience": coping_resilience(patient_id),
            "seizure_emotion": seizure_emotion_correlation(patient_id),
            "therapy_planning": therapy_planning(patient_id),
        },
    }


if __name__ == "__main__":
    r = full_dashboard()
    da = r["modules"]["depression_anxiety"]
    print("Psychologist dashboard:")
    print("  depression/anxiety:", da["summary"])
    print("  suicide-risk alerts:", len(da["alerts"]))
    print("  therapy plans:", r["modules"]["therapy_planning"]["priority_distribution"])
    print("  seizure-emotion corr:", r["modules"]["seizure_emotion"]["seizure_vs_depression_corr"])
