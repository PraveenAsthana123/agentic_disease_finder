"""
Speech-Language Pathologist (SLP) Module
========================================
Language and swallowing assessment analytics for epilepsy patients.

Endpoints:
  /api/slp                        — full dashboard (all 4 sub-analyses)
  /api/slp/language-assessment     — BNT + WAB aphasia quotient analysis
  /api/slp/speech-analysis         — verbal fluency (phonemic + semantic)
  /api/slp/swallowing              — MASA swallowing risk assessment
  /api/slp/pre-post-surgical       — pre/post-surgical language comparison

All data from REAL BNT, WAB, VERBAL_FLUENCY, MASA assessments in
data/clinical.db assessments table.
"""

import sqlite3
import os
import json
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _rows_as_dicts(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _parse_json(s):
    if not s:
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


# ─── 1. Language Assessment (BNT + WAB) ──────────────────────────────────

def language_assessment(patient_id=None):
    """Boston Naming Test (BNT) + Western Aphasia Battery (WAB) analysis.
    Returns per-patient naming + aphasia profiles with clinical flags."""
    c = _conn()

    where_clause = "AND a.patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()

    # BNT results
    bnt_rows = _rows_as_dicts(c.execute(
        f"SELECT a.*, p.age, p.gender FROM assessments a LEFT JOIN patients p ON a.patient_id = p.patient_id "
        f"WHERE a.instrument = 'BNT' {where_clause} ORDER BY a.patient_id",
        params
    ))

    # WAB results
    wab_rows = _rows_as_dicts(c.execute(
        f"SELECT a.*, p.age, p.gender FROM assessments a LEFT JOIN patients p ON a.patient_id = p.patient_id "
        f"WHERE a.instrument = 'WAB' {where_clause} ORDER BY a.patient_id",
        params
    ))
    c.close()

    # Build BNT summary
    bnt_summary = []
    bnt_levels = defaultdict(int)
    for r in bnt_rows:
        items = _parse_json(r["answers_json"])
        bnt_levels[r["level"]] += 1
        bnt_summary.append({
            "patient_id": r["patient_id"],
            "score": r["score"],
            "max_score": r["max_score"],
            "pct": round(100 * r["score"] / r["max_score"], 1) if r["max_score"] else 0,
            "level": r["level"],
            "interpretation": r["interpretation"],
            "semantic_cues_given": items.get("semantic_cues_given", 0),
            "phonemic_cues_given": items.get("phonemic_cues_given", 0),
            "semantic_cue_correct": items.get("semantic_cue_correct", 0),
            "phonemic_cue_correct": items.get("phonemic_cue_correct", 0),
            "alert": r["alert"],
            "age": r.get("age"),
            "gender": r.get("gender"),
        })

    bnt_level_dist = [{"level": k, "count": v, "pct": round(100 * v / len(bnt_rows), 1) if bnt_rows else 0}
                      for k, v in sorted(bnt_levels.items())]

    # Build WAB summary
    wab_summary = []
    wab_levels = defaultdict(int)
    aphasia_types = defaultdict(int)
    for r in wab_rows:
        items = _parse_json(r["answers_json"])
        wab_levels[r["level"]] += 1
        atype = items.get("aphasia_type", "Unclassified")
        aphasia_types[atype] += 1
        wab_summary.append({
            "patient_id": r["patient_id"],
            "aphasia_quotient": items.get("aphasia_quotient", r["score"]),
            "spontaneous_speech": items.get("spontaneous_speech"),
            "auditory_comprehension": items.get("auditory_comprehension"),
            "repetition": items.get("repetition"),
            "naming_word_finding": items.get("naming_word_finding"),
            "aphasia_type": atype,
            "level": r["level"],
            "interpretation": r["interpretation"],
            "alert": r["alert"],
        })

    wab_level_dist = [{"level": k, "count": v, "pct": round(100 * v / len(wab_rows), 1) if wab_rows else 0}
                      for k, v in sorted(wab_levels.items())]
    aphasia_type_dist = [{"type": k, "count": v} for k, v in sorted(aphasia_types.items(), key=lambda x: -x[1])]

    # Clinical flags: patients needing referral
    referral_needed = [b for b in bnt_summary if b["level"] in ("moderate", "severe")]
    aphasia_patients = [w for w in wab_summary if w["level"] in ("moderate", "severe")]

    avg_bnt = round(sum(b["score"] for b in bnt_summary) / len(bnt_summary), 1) if bnt_summary else 0
    avg_wab = round(sum(w["aphasia_quotient"] for w in wab_summary) / len(wab_summary), 1) if wab_summary else 0

    return {
        "total_patients": len(bnt_rows),
        "bnt": {
            "mean_score": avg_bnt,
            "max_score": 60,
            "level_distribution": bnt_level_dist,
            "patients_needing_referral": len(referral_needed),
            "per_patient": bnt_summary,
        },
        "wab": {
            "mean_aphasia_quotient": avg_wab,
            "level_distribution": wab_level_dist,
            "aphasia_type_distribution": aphasia_type_dist,
            "patients_with_aphasia": len(aphasia_patients),
            "per_patient": wab_summary,
        },
        "clinical_flags": {
            "naming_deficit_referral": [{"patient_id": r["patient_id"], "bnt_score": r["score"],
                                         "level": r["level"]} for r in referral_needed],
            "aphasia_therapy_needed": [{"patient_id": w["patient_id"], "aq": w["aphasia_quotient"],
                                        "type": w["aphasia_type"], "level": w["level"]}
                                       for w in aphasia_patients],
        }
    }


# ─── 2. Speech Analysis (Verbal Fluency) ──────────────────────────────────

def speech_analysis(patient_id=None):
    """Verbal fluency analysis: phonemic (FAS) + semantic (animals/fruits),
    with clustering/switching scores and executive-language flags."""
    c = _conn()

    where_clause = "AND a.patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()

    rows = _rows_as_dicts(c.execute(
        f"SELECT a.*, p.age, p.gender FROM assessments a LEFT JOIN patients p ON a.patient_id = p.patient_id "
        f"WHERE a.instrument = 'VERBAL_FLUENCY' {where_clause} ORDER BY a.patient_id",
        params
    ))
    c.close()

    if not rows:
        return {"total_patients": 0, "message": "No verbal fluency assessments found."}

    results = []
    levels = defaultdict(int)
    total_phon = 0
    total_sem = 0

    for r in rows:
        items = _parse_json(r["answers_json"])
        levels[r["level"]] += 1
        phon = items.get("phonemic_total", 0)
        sem = items.get("semantic_total", 0)
        total_phon += phon
        total_sem += sem

        results.append({
            "patient_id": r["patient_id"],
            "phonemic_total": phon,
            "phonemic_f": items.get("phonemic_f", 0),
            "phonemic_a": items.get("phonemic_a", 0),
            "phonemic_s": items.get("phonemic_s", 0),
            "semantic_total": sem,
            "semantic_animals": items.get("semantic_animals", 0),
            "semantic_fruits": items.get("semantic_fruits", 0),
            "perseverations": items.get("perseverations", 0),
            "intrusions": items.get("intrusions", 0),
            "clustering_score": items.get("clustering_score"),
            "switching_score": items.get("switching_score"),
            "total_score": r["score"],
            "level": r["level"],
            "interpretation": r["interpretation"],
            "alert": r["alert"],
            "age": r.get("age"),
            "gender": r.get("gender"),
        })

    n = len(results)
    level_dist = [{"level": k, "count": v, "pct": round(100 * v / n, 1)} for k, v in sorted(levels.items())]

    # Executive-language flags
    exec_flags = [r for r in results if r["level"] == "moderate"]
    high_persev = [r for r in results if r["perseverations"] >= 3]

    return {
        "total_patients": n,
        "mean_phonemic": round(total_phon / n, 1) if n else 0,
        "mean_semantic": round(total_sem / n, 1) if n else 0,
        "normal_cutoff_phonemic": 12,
        "normal_cutoff_semantic": 15,
        "level_distribution": level_dist,
        "executive_language_flags": len(exec_flags),
        "high_perseveration_count": len(high_persev),
        "per_patient": results,
        "clinical_flags": {
            "reduced_fluency_referral": [{"patient_id": r["patient_id"], "phonemic": r["phonemic_total"],
                                          "semantic": r["semantic_total"], "level": r["level"]}
                                         for r in exec_flags],
            "high_perseveration": [{"patient_id": r["patient_id"], "perseverations": r["perseverations"]}
                                   for r in high_persev],
        }
    }


# ─── 3. Swallowing Assessment (MASA) ──────────────────────────────────────

def swallowing_assessment(patient_id=None):
    """Modified Mann Assessment of Swallowing Ability (MASA).
    Evaluates aspiration risk — especially important post-ictally and
    for patients on sedating ASMs."""
    c = _conn()

    where_clause = "AND a.patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()

    rows = _rows_as_dicts(c.execute(
        f"SELECT a.*, p.age, p.gender FROM assessments a LEFT JOIN patients p ON a.patient_id = p.patient_id "
        f"WHERE a.instrument = 'MASA' {where_clause} ORDER BY a.patient_id",
        params
    ))
    c.close()

    if not rows:
        return {"total_patients": 0, "message": "No MASA assessments found."}

    results = []
    levels = defaultdict(int)
    total_score = 0

    for r in rows:
        items = _parse_json(r["answers_json"])
        levels[r["level"]] += 1
        total_score += r["score"]

        results.append({
            "patient_id": r["patient_id"],
            "masa_score": r["score"],
            "max_score": r["max_score"],
            "level": r["level"],
            "interpretation": r["interpretation"],
            "alertness": items.get("alertness"),
            "cooperation": items.get("cooperation"),
            "respiration": items.get("respiration"),
            "oral_motor": items.get("oral_motor"),
            "tongue_movement": items.get("tongue_movement"),
            "gag_reflex": items.get("gag_reflex"),
            "voluntary_cough": items.get("voluntary_cough"),
            "palate_movement": items.get("palate_movement"),
            "bolus_clearance": items.get("bolus_clearance"),
            "pharyngeal_response": items.get("pharyngeal_response"),
            "post_ictal_risk_flag": items.get("post_ictal_risk_flag", False),
            "rescue_med_aspiration_risk": items.get("rescue_med_aspiration_risk", False),
            "alert": r["alert"],
            "age": r.get("age"),
            "gender": r.get("gender"),
        })

    n = len(results)
    level_dist = [{"level": k, "count": v, "pct": round(100 * v / n, 1)} for k, v in sorted(levels.items())]

    # Aspiration risk patients
    aspiration_risk = [r for r in results if r["level"] in ("moderate", "severe")]
    post_ictal_risk = [r for r in results if r["post_ictal_risk_flag"]]
    rescue_med_risk = [r for r in results if r["rescue_med_aspiration_risk"]]

    return {
        "total_patients": n,
        "mean_masa_score": round(total_score / n, 1) if n else 0,
        "max_score": 200,
        "normal_cutoff": 170,
        "level_distribution": level_dist,
        "aspiration_risk_count": len(aspiration_risk),
        "post_ictal_risk_count": len(post_ictal_risk),
        "rescue_med_risk_count": len(rescue_med_risk),
        "per_patient": results,
        "clinical_flags": {
            "dysphagia_referral": [{"patient_id": r["patient_id"], "masa_score": r["masa_score"],
                                    "level": r["level"]} for r in aspiration_risk],
            "post_ictal_aspiration_risk": [{"patient_id": r["patient_id"],
                                            "masa_score": r["masa_score"]} for r in post_ictal_risk],
            "rescue_med_aspiration": [{"patient_id": r["patient_id"]} for r in rescue_med_risk],
        }
    }


# ─── 4. Pre/Post-Surgical Language Comparison ────────────────────────────

def pre_post_surgical(patient_id=None):
    """Compare language scores across assessment dates to detect
    post-surgical or post-treatment language changes.

    Uses BNT + WAB + VERBAL_FLUENCY scores grouped by patient,
    ordered by date. If a patient has multiple assessments, the
    earliest is treated as 'pre' and the latest as 'post'.

    For patients with single assessments, provides baseline + risk estimate
    based on lateralization factors (WAB sub-scores).
    """
    c = _conn()

    where_clause = "AND a.patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()

    rows = _rows_as_dicts(c.execute(
        f"SELECT a.*, p.age, p.gender FROM assessments a LEFT JOIN patients p ON a.patient_id = p.patient_id "
        f"WHERE a.instrument IN ('BNT','WAB','VERBAL_FLUENCY') {where_clause} "
        f"ORDER BY a.patient_id, a.instrument, a.created_at",
        params
    ))
    c.close()

    # Group by patient
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    comparisons = []
    for pid, records in sorted(by_patient.items()):
        by_inst = defaultdict(list)
        for r in records:
            by_inst[r["instrument"]].append(r)

        patient_comp = {
            "patient_id": pid,
            "age": records[0].get("age"),
            "gender": records[0].get("gender"),
            "instruments_assessed": list(by_inst.keys()),
            "has_multiple_timepoints": False,
            "baselines": {},
            "surgical_risk_estimate": None,
        }

        # For each instrument, get baseline scores
        for inst, inst_rows in by_inst.items():
            first = inst_rows[0]
            items = _parse_json(first["answers_json"])

            if inst == "BNT":
                patient_comp["baselines"]["bnt_score"] = first["score"]
                patient_comp["baselines"]["bnt_level"] = first["level"]
            elif inst == "WAB":
                patient_comp["baselines"]["wab_aq"] = items.get("aphasia_quotient", first["score"])
                patient_comp["baselines"]["wab_level"] = first["level"]
                patient_comp["baselines"]["aphasia_type"] = items.get("aphasia_type", "Unknown")
                # Estimate surgical risk from WAB sub-scores
                sp = items.get("spontaneous_speech", 0)
                comp = items.get("auditory_comprehension", 0)
                nam = items.get("naming_word_finding", 0)
                # Left-lateralized TLE patients with lower naming/fluency scores
                # are at higher risk for post-surgical language decline
                risk_score = round(10 - (sp / 2 + comp + nam) / 4, 1)
                risk_level = "high" if risk_score > 5 else "moderate" if risk_score > 3 else "low"
                patient_comp["surgical_risk_estimate"] = {
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "key_factors": {
                        "spontaneous_speech": sp,
                        "comprehension": comp,
                        "naming": nam,
                    },
                    "recommendation": (
                        "Pre-surgical Wada test + fMRI language mapping strongly recommended"
                        if risk_level == "high" else
                        "Standard pre-surgical language assessment recommended"
                        if risk_level == "moderate" else
                        "Routine pre-surgical evaluation sufficient"
                    )
                }
            elif inst == "VERBAL_FLUENCY":
                patient_comp["baselines"]["vf_phonemic"] = items.get("phonemic_total", 0)
                patient_comp["baselines"]["vf_semantic"] = items.get("semantic_total", 0)
                patient_comp["baselines"]["vf_level"] = first["level"]

            # Check for multiple timepoints
            if len(inst_rows) > 1:
                patient_comp["has_multiple_timepoints"] = True
                last = inst_rows[-1]
                last_items = _parse_json(last["answers_json"])
                delta = round(last["score"] - first["score"], 1)
                patient_comp.setdefault("changes", {})[inst] = {
                    "pre_score": first["score"],
                    "post_score": last["score"],
                    "delta": delta,
                    "pre_level": first["level"],
                    "post_level": last["level"],
                    "pre_date": first["created_at"],
                    "post_date": last["created_at"],
                    "clinically_significant": abs(delta) > (first["max_score"] * 0.1),
                }

        comparisons.append(patient_comp)

    # Summary stats
    high_risk = [c for c in comparisons if c.get("surgical_risk_estimate", {}).get("risk_level") == "high"]
    with_changes = [c for c in comparisons if c.get("has_multiple_timepoints")]

    return {
        "total_patients": len(comparisons),
        "patients_with_multiple_timepoints": len(with_changes),
        "high_surgical_risk_count": len(high_risk),
        "per_patient": comparisons,
        "clinical_flags": {
            "high_surgical_risk": [{"patient_id": c["patient_id"],
                                     "risk_score": c["surgical_risk_estimate"]["risk_score"],
                                     "recommendation": c["surgical_risk_estimate"]["recommendation"]}
                                    for c in high_risk],
        }
    }


# ─── Full Dashboard ──────────────────────────────────────────────────────

def full_dashboard(patient_id=None):
    """Full SLP dashboard: language assessment + speech analysis + swallowing + pre/post-surgical."""
    lang = language_assessment(patient_id)
    speech = speech_analysis(patient_id)
    swallow = swallowing_assessment(patient_id)
    prepost = pre_post_surgical(patient_id)

    # Aggregate clinical urgency
    urgent_patients = set()
    for flag_group in [lang.get("clinical_flags", {}), speech.get("clinical_flags", {}),
                       swallow.get("clinical_flags", {}), prepost.get("clinical_flags", {})]:
        for flag_name, flag_list in flag_group.items():
            for f in flag_list:
                urgent_patients.add(f.get("patient_id"))

    return {
        "role": "Speech-Language Pathologist (SLP)",
        "total_patients_assessed": lang.get("total_patients", 0),
        "urgent_referrals": len(urgent_patients),
        "urgent_patient_ids": sorted(urgent_patients),
        "summary": {
            "bnt_mean": lang.get("bnt", {}).get("mean_score"),
            "wab_mean_aq": lang.get("wab", {}).get("mean_aphasia_quotient"),
            "vf_mean_phonemic": speech.get("mean_phonemic"),
            "vf_mean_semantic": speech.get("mean_semantic"),
            "masa_mean": swallow.get("mean_masa_score"),
            "naming_deficit_count": lang.get("bnt", {}).get("patients_needing_referral", 0),
            "aphasia_count": lang.get("wab", {}).get("patients_with_aphasia", 0),
            "dysphagia_risk_count": swallow.get("aspiration_risk_count", 0),
            "high_surgical_risk": prepost.get("high_surgical_risk_count", 0),
        },
        "language_assessment": lang,
        "speech_analysis": speech,
        "swallowing": swallow,
        "pre_post_surgical": prepost,
    }
