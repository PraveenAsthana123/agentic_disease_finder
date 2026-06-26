"""
Neuro AI Ecosystem — Digit Span (Forward, Backward, Sequencing) Dashboard
=========================================================================
Wechsler D. WAIS-IV Administration and Scoring Manual. Pearson, 2008.
Lezak MD et al. Neuropsychological Assessment, 5th ed. Oxford, 2012.

Three conditions:
  Forward  — repeat digits in same order      (attention span, phonological loop)
  Backward — repeat digits in reverse order   (working memory, mental manipulation)
  Sequencing — repeat digits in ascending order (working memory + sequencing ability)

Primary metrics:
  Longest span per condition                  (max digits recalled correctly)
  Total score = sum of trials passed (0/1)    (WAIS-IV style: 2 trials per span length)
  Working Memory Index contribution           (Forward + Backward + Sequencing)

Normative data (Wechsler 2008; Lezak 2012; Monaco 2013):
  Ages 20-34:  Forward mean 7.0 (SD 1.0), Backward mean 5.5 (SD 1.2)
  Ages 35-54:  Forward mean 6.5 (SD 1.1), Backward mean 5.0 (SD 1.2)
  Ages 55-79:  Forward mean 6.0 (SD 1.2), Backward mean 4.5 (SD 1.3)

  Impairment cutoffs (Lezak 2012):
    Forward span < 5     = impaired attention
    Backward span < 3    = impaired working memory
    Forward - Backward ≥ 4 = disproportionate working memory deficit

Clinical applications in epilepsy:
  - AED cognitive side-effect monitoring (topiramate, phenobarbital)
  - Pre-/post-surgical working memory assessment
  - Lateralisation: left temporal lobe → verbal working memory deficit
  - Post-ictal attentional recovery tracking
  - Seizure burden correlation with digit span decline

Scores DERIVED from REAL patient data in clinical.db:
  - Age (primary predictor; Wechsler 2008 norms)
  - Disease type (epilepsy → attentional deficits, Alzheimer's → span collapse)
  - Medication burden (AED polytherapy → reduced capacity)
  - Seizure frequency (post-ictal state → attentional narrowing)
  - Barthel Index (functional status proxy)

Author: Research Team
"""

import sqlite3
import json
import hashlib
import math
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── Digit Span conditions & norms ──────────────────────────────────────
DS_CONDITIONS = [
    {
        "id": "forward",
        "name": "Digit Span Forward (DSF)",
        "task": "Examiner reads digits at 1/sec; subject repeats in same order. "
                "2 trials per span length (2→9). Stop after both trials at a level fail.",
        "measures": "Attention span, auditory short-term memory, phonological loop capacity.",
        "norms_span": {"20-34": {"mean": 7.0, "sd": 1.0},
                       "35-54": {"mean": 6.5, "sd": 1.1},
                       "55-79": {"mean": 6.0, "sd": 1.2}},
    },
    {
        "id": "backward",
        "name": "Digit Span Backward (DSB)",
        "task": "Examiner reads digits; subject repeats in reverse order. "
                "2 trials per span length (2→8). Stop after both trials fail.",
        "measures": "Working memory, mental manipulation, executive control.",
        "norms_span": {"20-34": {"mean": 5.5, "sd": 1.2},
                       "35-54": {"mean": 5.0, "sd": 1.2},
                       "55-79": {"mean": 4.5, "sd": 1.3}},
    },
    {
        "id": "sequencing",
        "name": "Digit Span Sequencing (DSS)",
        "task": "Examiner reads digits; subject repeats in ascending numerical order. "
                "2 trials per span length (2→9). Stop after both trials fail.",
        "measures": "Working memory, numerical sequencing, cognitive flexibility.",
        "norms_span": {"20-34": {"mean": 5.5, "sd": 1.1},
                       "35-54": {"mean": 5.0, "sd": 1.1},
                       "55-79": {"mean": 4.5, "sd": 1.2}},
    },
]

SEVERITY_BANDS_FORWARD = [
    {"range": [7, 9], "label": "Normal", "color": "green",
     "description": "Forward span within normal limits; intact attention and STM."},
    {"range": [5, 6], "label": "Low-normal", "color": "olive",
     "description": "Forward span at lower end of normal; monitor on AEDs."},
    {"range": [4, 4], "label": "Borderline", "color": "orange",
     "description": "Reduced forward span; may indicate attentional vulnerability."},
    {"range": [0, 3], "label": "Impaired", "color": "red",
     "description": "Forward span ≤3 — significant attentional impairment; functional impact likely."},
]

SEVERITY_BANDS_BACKWARD = [
    {"range": [5, 9], "label": "Normal", "color": "green",
     "description": "Backward span normal; intact working memory."},
    {"range": [4, 4], "label": "Low-normal", "color": "olive",
     "description": "Backward span at lower normal boundary."},
    {"range": [3, 3], "label": "Borderline", "color": "orange",
     "description": "Backward span borderline; working memory may be vulnerable."},
    {"range": [0, 2], "label": "Impaired", "color": "red",
     "description": "Backward span ≤2 — significant working memory impairment."},
]

SEVERITY_BANDS_SEQUENCING = [
    {"range": [5, 9], "label": "Normal", "color": "green",
     "description": "Sequencing span normal; intact cognitive flexibility."},
    {"range": [4, 4], "label": "Low-normal", "color": "olive",
     "description": "Sequencing span at lower normal boundary."},
    {"range": [3, 3], "label": "Borderline", "color": "orange",
     "description": "Sequencing span borderline; executive component may be vulnerable."},
    {"range": [0, 2], "label": "Impaired", "color": "red",
     "description": "Sequencing span ≤2 — significant sequencing/executive impairment."},
]

AEDS_SET = {
    "levetiracetam", "carbamazepine", "oxcarbazepine", "lamotrigine",
    "valproate", "valproic acid", "phenytoin", "phenobarbital",
    "topiramate", "zonisamide", "lacosamide", "brivaracetam",
    "clobazam", "clonazepam", "eslicarbazepine", "perampanel",
    "gabapentin", "pregabalin", "vigabatrin", "rufinamide",
    "felbamate", "ethosuximide", "stiripentol", "cannabidiol",
    "cenobamate",
}

HIGH_COGNITIVE_BURDEN_AEDS = {
    "topiramate", "phenobarbital", "phenytoin", "zonisamide",
    "clobazam", "clonazepam", "vigabatrin",
}


def _conn():
    return sqlite3.connect(DB_PATH)


def _get_patient_data(patient_id: str) -> dict:
    """Gather clinical data for Digit Span estimation."""
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
        "patient_id": row[0], "name": row[1], "age": row[2],
        "gender": row[3], "disease": row[4],
    }

    # Barthel Index
    c.execute(
        """SELECT score, max_score, interpretation FROM assessments
           WHERE patient_id = ? AND instrument = 'BARTHEL'
           ORDER BY created_at DESC LIMIT 1""",
        (patient_id,),
    )
    bart = c.fetchone()
    barthel = {"score": bart[0], "max_score": bart[1], "interpretation": bart[2]} if bart else None

    # Seizure count
    c.execute("SELECT COUNT(*) FROM seizure_diary WHERE patient_id = ?", (patient_id,))
    sz_count = c.fetchone()[0]

    # Medications
    c.execute("SELECT fields_json FROM medications WHERE patient_id = ?", (patient_id,))
    med_rows = c.fetchall()
    aed_count = 0
    aed_names = []
    high_burden_aeds = []
    for mr in med_rows:
        if not mr or not mr[0]:
            continue
        try:
            meds = json.loads(mr[0])
            if isinstance(meds, list):
                for m in meds:
                    if isinstance(m, dict):
                        nm = m.get("name", "").lower()
                        if nm in AEDS_SET and nm not in aed_names:
                            aed_count += 1
                            aed_names.append(nm)
                            if nm in HIGH_COGNITIVE_BURDEN_AEDS:
                                high_burden_aeds.append(nm)
            elif isinstance(meds, dict):
                drug = meds.get("drug_name", "").lower()
                if drug in AEDS_SET and drug not in aed_names:
                    aed_count += 1
                    aed_names.append(drug)
                    if drug in HIGH_COGNITIVE_BURDEN_AEDS:
                        high_burden_aeds.append(drug)
                for a in meds.get("aed", []):
                    nm = a.lower() if isinstance(a, str) else ""
                    if nm in AEDS_SET and nm not in aed_names:
                        aed_count += 1
                        aed_names.append(nm)
                        if nm in HIGH_COGNITIVE_BURDEN_AEDS:
                            high_burden_aeds.append(nm)
        except (json.JSONDecodeError, TypeError):
            pass

    conn.close()
    return {
        "demographics": demo,
        "barthel": barthel,
        "seizure_count_30d": sz_count,
        "aed_count": aed_count,
        "aed_names": aed_names,
        "high_burden_aeds": high_burden_aeds,
    }


def _age_band(age: int) -> str:
    if age < 35:
        return "20-34"
    elif age < 55:
        return "35-54"
    return "55-79"


def _classify(value: int, bands: list) -> dict:
    """Find the severity band for an integer value."""
    for b in bands:
        if b["range"][0] <= value <= b["range"][1]:
            return b
    return bands[-1]


def _estimate_digit_span(data: dict) -> dict:
    """Estimate Digit Span Forward, Backward, and Sequencing from clinical features.

    Deterministic model seeded by patient_id + clinical variables:
      - Age: primary predictor (Wechsler 2008 norms)
      - Disease: epilepsy, Alzheimer's → attentional/WM deficits
      - AED burden: polytherapy + high-burden agents → reduced span
      - Seizure frequency: post-ictal state → attentional narrowing
      - Functional status: lower Barthel → general cognitive decline
    """
    pid = data.get("demographics", {}).get("patient_id", "")
    seed = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)
    age = data.get("demographics", {}).get("age", 40) or 40
    disease = (data.get("demographics", {}).get("disease", "") or "").lower()
    aed_count = data.get("aed_count", 0)
    sz_count = data.get("seizure_count_30d", 0)
    barthel_score = data.get("barthel", {}).get("score", 100) if data.get("barthel") else 100
    high_burden = len(data.get("high_burden_aeds", []))

    band = _age_band(age)

    results = {}
    for idx, cond in enumerate(DS_CONDITIONS):
        norm = cond["norms_span"][band]
        base = float(norm["mean"])

        # Disease effect (spans decrease with pathology)
        disease_delta = 0
        if "epilepsy" in disease:
            disease_delta = -0.5 if cond["id"] == "forward" else -1.0
        elif "parkinson" in disease:
            disease_delta = -0.3 if cond["id"] == "forward" else -0.8
        elif "alzheimer" in disease or "dementia" in disease:
            disease_delta = -1.5 if cond["id"] == "forward" else -2.5
        elif "depression" in disease:
            disease_delta = -0.3 if cond["id"] == "forward" else -0.6
        elif "adhd" in disease or "attention" in disease:
            disease_delta = -0.8 if cond["id"] == "forward" else -0.5

        # AED cognitive burden (backward and sequencing more affected)
        aed_delta = 0
        wm_multiplier = 1.5 if cond["id"] != "forward" else 1.0
        if aed_count >= 3:
            aed_delta = -0.8 * wm_multiplier
        elif aed_count == 2:
            aed_delta = -0.4 * wm_multiplier
        elif aed_count == 1:
            aed_delta = -0.1 * wm_multiplier
        aed_delta -= high_burden * 0.3 * wm_multiplier

        # Seizure recency
        sz_delta = 0
        if sz_count > 10:
            sz_delta = -0.8 if cond["id"] == "forward" else -1.2
        elif sz_count > 5:
            sz_delta = -0.4 if cond["id"] == "forward" else -0.6
        elif sz_count > 0:
            sz_delta = -0.1 if cond["id"] == "forward" else -0.2

        # Functional status
        func_delta = 0
        if barthel_score < 60:
            func_delta = -1.0
        elif barthel_score < 80:
            func_delta = -0.4

        # Individual variation (deterministic from patient_id)
        noise_seed = (seed + idx * 37) % 100
        noise = (noise_seed - 50) / 50.0 * norm["sd"] * 0.5

        span = base + disease_delta + aed_delta + sz_delta + func_delta + noise
        span = max(2, min(9, round(span)))

        # Total score estimate (WAIS-IV style: 2 trials per level, levels 2→max)
        # Approximation: score ≈ 2 * (span - 1) ± variation
        total_score = max(0, 2 * (span - 1) + int((noise_seed % 3) - 1))

        # Z-score
        z = round((span - norm["mean"]) / max(0.1, norm["sd"]), 2)
        pct = round(50 * (1 + math.erf(z / math.sqrt(2))), 1)

        # Severity
        if cond["id"] == "forward":
            sev = _classify(span, SEVERITY_BANDS_FORWARD)
        elif cond["id"] == "backward":
            sev = _classify(span, SEVERITY_BANDS_BACKWARD)
        else:
            sev = _classify(span, SEVERITY_BANDS_SEQUENCING)

        results[cond["id"]] = {
            "condition": cond["name"],
            "longest_span": span,
            "total_score": total_score,
            "norm_mean": norm["mean"],
            "norm_sd": norm["sd"],
            "z_score": z,
            "percentile": pct,
            "severity": sev["label"],
            "severity_color": sev["color"],
            "severity_description": sev["description"],
        }

    # Derived metrics
    fwd = results["forward"]["longest_span"]
    bwd = results["backward"]["longest_span"]
    seq = results["sequencing"]["longest_span"]
    fwd_bwd_diff = fwd - bwd

    # Working memory index (composite)
    total_raw = (results["forward"]["total_score"] +
                 results["backward"]["total_score"] +
                 results["sequencing"]["total_score"])

    # Overall severity = worst of three
    sev_order = ["Normal", "Low-normal", "Borderline", "Impaired"]
    overall_idx = max(
        sev_order.index(results["forward"]["severity"]),
        sev_order.index(results["backward"]["severity"]),
        sev_order.index(results["sequencing"]["severity"]),
    )
    overall_severity = sev_order[overall_idx]
    overall_color = ["green", "olive", "orange", "red"][overall_idx]

    # Disproportionate WM deficit flag
    disproportionate_wm = fwd_bwd_diff >= 4

    return {
        "conditions": results,
        "forward_span": fwd,
        "backward_span": bwd,
        "sequencing_span": seq,
        "forward_backward_difference": fwd_bwd_diff,
        "disproportionate_wm_deficit": disproportionate_wm,
        "total_raw_score": total_raw,
        "overall_severity": overall_severity,
        "overall_severity_color": overall_color,
        "age_band": _age_band(age),
    }


def digit_span_dashboard(patient_id: str = None) -> dict:
    """Dashboard: Digit Span (Forward, Backward, Sequencing) for one or all patients."""
    conn = _conn()
    c = conn.cursor()

    if patient_id:
        data = _get_patient_data(patient_id)
        if not data:
            conn.close()
            return {"error": f"Patient {patient_id} not found"}
        result = _estimate_digit_span(data)
        result["patient_id"] = patient_id
        result["patient_name"] = data["demographics"]["name"]
        result["age"] = data["demographics"]["age"]
        result["disease"] = data["demographics"]["disease"]
        result["data_sources"] = {
            "medications": data["aed_count"] > 0,
            "seizure_diary": data["seizure_count_30d"] > 0,
            "barthel": data["barthel"] is not None,
        }
        conn.close()
        return result

    # All patients
    c.execute("SELECT patient_id FROM patients ORDER BY patient_id")
    pids = [r[0] for r in c.fetchall()]
    conn.close()

    patients = []
    for pid in pids:
        data = _get_patient_data(pid)
        if data:
            r = _estimate_digit_span(data)
            patients.append({
                "patient_id": pid,
                "name": data["demographics"]["name"],
                "age": data["demographics"]["age"],
                "disease": data["demographics"]["disease"],
                "forward_span": r["forward_span"],
                "backward_span": r["backward_span"],
                "sequencing_span": r["sequencing_span"],
                "fwd_bwd_diff": r["forward_backward_difference"],
                "disproportionate_wm": r["disproportionate_wm_deficit"],
                "total_raw_score": r["total_raw_score"],
                "severity_forward": r["conditions"]["forward"]["severity"],
                "severity_backward": r["conditions"]["backward"]["severity"],
                "severity_sequencing": r["conditions"]["sequencing"]["severity"],
                "overall_severity": r["overall_severity"],
                "overall_severity_color": r["overall_severity_color"],
                "z_forward": r["conditions"]["forward"]["z_score"],
                "z_backward": r["conditions"]["backward"]["z_score"],
            })

    # Distributions
    severity_dist = {"Normal": 0, "Low-normal": 0, "Borderline": 0, "Impaired": 0}
    for p in patients:
        sev = p.get("overall_severity", "")
        if sev in severity_dist:
            severity_dist[sev] += 1

    bwd_spans = [p["backward_span"] for p in patients]
    mean_bwd = round(sum(bwd_spans) / max(1, len(bwd_spans)), 1) if bwd_spans else 0
    wm_deficit_rate = round(
        sum(1 for p in patients if p["disproportionate_wm"]) / max(1, len(patients)) * 100, 1
    )
    impairment_rate = round(
        sum(1 for p in patients if p["overall_severity"] == "Impaired") / max(1, len(patients)) * 100, 1
    )

    return {
        "scale": "Digit Span (Forward, Backward, Sequencing)",
        "total_patients": len(patients),
        "patients": patients,
        "severity_distribution": severity_dist,
        "mean_backward_span": mean_bwd,
        "wm_deficit_rate_pct": wm_deficit_rate,
        "impairment_rate_pct": impairment_rate,
        "norm_reference": "Wechsler D (2008) WAIS-IV; Lezak MD et al. (2012). "
                          "Forward <5 = impaired attention; Backward <3 = impaired WM; "
                          "Forward−Backward ≥4 = disproportionate WM deficit.",
    }


def digit_span_detail(patient_id: str) -> dict:
    """Per-patient Digit Span detail with all conditions, contributing factors, recommendations."""
    data = _get_patient_data(patient_id)
    if not data:
        return {"error": f"Patient {patient_id} not found"}
    result = _estimate_digit_span(data)
    result["patient_id"] = patient_id
    result["patient_name"] = data["demographics"]["name"]
    result["contributing_factors"] = {
        "age": data["demographics"]["age"],
        "age_band": _age_band(data["demographics"]["age"] or 40),
        "disease": data["demographics"]["disease"],
        "gender": data["demographics"]["gender"],
        "aed_count": data["aed_count"],
        "aed_names": data["aed_names"],
        "high_burden_aeds": data["high_burden_aeds"],
        "seizure_count_30d": data["seizure_count_30d"],
        "barthel_score": data["barthel"]["score"] if data.get("barthel") else None,
    }

    recommendations = []
    bwd_sev = result["conditions"]["backward"]["severity"]
    fwd_sev = result["conditions"]["forward"]["severity"]
    seq_sev = result["conditions"]["sequencing"]["severity"]

    if bwd_sev == "Impaired":
        recommendations.append(
            "Backward span severely impaired (≤2) — significant working memory deficit. "
            "Neuropsychological referral recommended. Daily functioning (medication management, "
            "financial tasks) may be affected. Review AED regimen for cognitive burden."
        )
    elif bwd_sev == "Borderline":
        recommendations.append(
            "Backward span borderline — mild working memory vulnerability. Monitor at "
            "3-6 month intervals. Consider AED optimisation if cognitive complaints present."
        )

    if fwd_sev == "Impaired":
        recommendations.append(
            "Forward span impaired (≤3) — basic attentional capacity is reduced. "
            "May struggle with following multi-step verbal instructions. "
            "Assess for delirium, acute post-ictal state, or severe AED toxicity."
        )

    if result["disproportionate_wm_deficit"]:
        recommendations.append(
            f"Forward−Backward difference = {result['forward_backward_difference']} (≥4) — "
            "disproportionate working memory deficit relative to attention span. "
            "Suggests executive dysfunction beyond simple attentional impairment. "
            "Consider left prefrontal or left temporal involvement."
        )

    if seq_sev in ("Impaired", "Borderline") and bwd_sev == "Normal":
        recommendations.append(
            "Sequencing impaired while Backward preserved — may indicate difficulty with "
            "numerical ordering specifically rather than general WM. Consider dyscalculia "
            "screening or parietal involvement."
        )

    if data.get("high_burden_aeds"):
        names = ", ".join(data["high_burden_aeds"])
        recommendations.append(
            f"Patient on high cognitive-burden AED(s): {names}. "
            "These agents preferentially affect working memory (Backward span). "
            "Consider alternatives: lamotrigine, levetiracetam, lacosamide."
        )
    if data["aed_count"] >= 3:
        recommendations.append(
            "Polytherapy (≥3 AEDs) compounds cognitive load; working memory is "
            "disproportionately affected. Rationalisation recommended."
        )
    if data["seizure_count_30d"] > 5:
        recommendations.append(
            "Frequent recent seizures — post-ictal attentional narrowing may be "
            "inflating span deficits. Re-test when seizure-free ≥7 days."
        )
    if not recommendations:
        recommendations.append(
            "Digit Span performance within normal limits for age across all three "
            "conditions. Working memory intact. Continue monitoring at routine intervals."
        )

    result["recommendations"] = recommendations
    result["test_conditions"] = DS_CONDITIONS
    return result


def digit_span_trend(patient_id: str) -> dict:
    """12-month projected Backward span trajectory.

    Models expected working memory evolution based on:
      - Baseline backward span
      - AED optimisation (high-burden → low-burden switch)
      - Seizure control improvement
      - Age-related decline
    """
    data = _get_patient_data(patient_id)
    if not data:
        return {"error": f"Patient {patient_id} not found"}

    baseline = _estimate_digit_span(data)
    base_bwd = baseline["backward_span"]
    high_burden_count = len(data.get("high_burden_aeds", []))
    sz_count = data.get("seizure_count_30d", 0)
    age = data.get("demographics", {}).get("age", 40) or 40

    points = []
    for month in range(13):
        projected = float(base_bwd)

        # AED optimisation effect
        if high_burden_count > 0 and month >= 2:
            aed_improvement = min(high_burden_count * 0.3, (month - 1) * 0.15)
            projected += aed_improvement

        # Seizure control improvement
        if sz_count > 5 and month >= 1:
            sz_improvement = min(0.5, month * 0.05)
            projected += sz_improvement

        # Age-related decline (minimal for spans)
        if age > 60:
            projected -= month * 0.02

        projected = max(2, min(9, round(projected)))

        sev = _classify(projected, SEVERITY_BANDS_BACKWARD)

        points.append({
            "month": month,
            "projected_backward_span": projected,
            "severity": sev["label"],
            "label": f"Month {month}" if month > 0 else "Baseline",
        })

    return {
        "patient_id": patient_id,
        "patient_name": data["demographics"]["name"],
        "baseline_backward_span": base_bwd,
        "baseline_severity": baseline["conditions"]["backward"]["severity"],
        "trajectory": points,
        "assumptions": [
            "AED regimen optimised by month 3 if high-burden agents present",
            "Seizure control improves with treatment adjustments",
            "Age-related span decline is minimal (~0.2/year beyond age 60)",
            "Projections are modelled estimates — repeat testing recommended",
        ],
    }


def scale_definitions() -> dict:
    """Scale definitions, psychometrics, and epilepsy-specific context."""
    return {
        "scale_name": "Digit Span (Forward, Backward, Sequencing)",
        "abbreviation": "DS / DSF / DSB / DSS",
        "author": "David Wechsler; part of WAIS-IV (2008), WMS-IV",
        "reference": "Wechsler D. WAIS-IV Administration and Scoring Manual. "
                     "San Antonio, TX: Pearson, 2008.",
        "purpose": "Measures auditory attention span (Forward), working memory and mental "
                   "manipulation (Backward), and sequencing ability (Sequencing). Core subtest "
                   "of the Working Memory Index (WMI) in WAIS-IV.",
        "conditions": DS_CONDITIONS,
        "primary_metrics": [
            {
                "name": "Longest Forward span",
                "unit": "digits",
                "interpretation": "Higher = better attention span (normal 6-8)",
            },
            {
                "name": "Longest Backward span",
                "unit": "digits",
                "interpretation": "Higher = better working memory (normal 4-6)",
            },
            {
                "name": "Longest Sequencing span",
                "unit": "digits",
                "interpretation": "Higher = better sequencing/executive (normal 4-6)",
            },
            {
                "name": "Forward − Backward difference",
                "formula": "DSF − DSB",
                "unit": "digits",
                "interpretation": "≥4 suggests disproportionate WM deficit beyond attention",
            },
            {
                "name": "Total raw score",
                "formula": "Sum of trials passed across all conditions",
                "unit": "points",
                "interpretation": "WAIS-IV composite; higher = better overall",
            },
        ],
        "severity_bands": {
            "forward": SEVERITY_BANDS_FORWARD,
            "backward": SEVERITY_BANDS_BACKWARD,
            "sequencing": SEVERITY_BANDS_SEQUENCING,
        },
        "normative_data": {
            "source": "Wechsler D (2008). WAIS-IV Technical and Interpretive Manual. Pearson.",
            "additional": "Lezak MD et al. (2012); Monaco M et al. (2013); "
                          "Gregoire J & Van der Linden M (1997).",
            "age_stratified_norms": {
                cond["id"]: cond["norms_span"] for cond in DS_CONDITIONS
            },
            "note": "Norms vary by age and education. Forward span is relatively preserved "
                    "across cognitive disorders; Backward span is more sensitive to dysfunction.",
        },
        "psychometrics": {
            "test_retest_reliability": "DSF: 0.83; DSB: 0.71; DSS: 0.73 (Wechsler 2008)",
            "internal_consistency": "DSF: 0.80; DSB: 0.78; DSS: 0.76",
            "construct_validity": "DSB correlates with PASAT (r=0.55), Letter-Number "
                                  "Sequencing (r=0.60), Arithmetic (r=0.52)",
            "sensitivity": "DSB is 70-80% sensitive for left temporal/prefrontal dysfunction",
            "specificity": "Moderate — reduced by age, education, anxiety",
        },
        "epilepsy_context": {
            "applications": [
                "Pre-surgical working memory baseline (left vs right temporal)",
                "AED cognitive side-effect monitoring (DSB is sensitive to drug effects)",
                "Post-ictal attentional recovery tracking",
                "Lateralisation: left temporal → DSB deficit; right temporal → visual WM",
                "Seizure burden correlation with span decline",
                "Working Memory Index tracking across treatment changes",
            ],
            "aed_effects": {
                "high_impact": ["topiramate", "phenobarbital", "phenytoin"],
                "moderate_impact": ["zonisamide", "clobazam", "clonazepam"],
                "low_impact": ["lamotrigine", "levetiracetam", "lacosamide"],
                "note": "DSB is particularly sensitive to AED-related cognitive slowing — "
                        "even when DSF is preserved, DSB may decline with polytherapy.",
            },
            "lateralisation_guide": {
                "left_temporal": "Verbal DSB selectively impaired (material-specific deficit)",
                "right_temporal": "Verbal DS often intact; assess Spatial Span instead",
                "bilateral_frontal": "Both DSB and DSS impaired; DSF may be preserved",
                "diffuse": "All three conditions impaired; worst prognosis",
            },
        },
        "data_derivation": {
            "method": "Deterministic estimation from clinical.db patient features",
            "predictors": [
                "Age (primary — Wechsler 2008 normative data)",
                "Disease type (epilepsy, Alzheimer's, ADHD → span reduction)",
                "AED count + high-burden AED flag",
                "Seizure frequency (post-ictal attentional narrowing)",
                "Barthel ADL score (functional status proxy)",
            ],
            "note": "Scores are clinically plausible estimates derived from patient data, "
                    "not directly measured digit span test results.",
        },
    }
