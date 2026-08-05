"""Agent Grounding Gate Dashboard — hallucination detection via expert agreement.

Addresses two P1 production issues from config/production_issues.json:
  1. Agent Hallucination     — high-confidence predictions that experts reject
  2. Grounding/Citation Failure — AI outputs not anchored to verifiable evidence

Clinically this matters because:
- AI predictions with high confidence but expert disagreement indicate
  grounding failure — the model is 'hallucinating' a diagnosis not
  supported by EEG features (Rajpurkar et al., Nat Med 2022).
- Calibration error (ECE) measures the gap between stated confidence and
  actual accuracy — well-calibrated models are safer for clinical use
  (Guo et al., ICML 2017).
- HITL override rate tracks cases where human reviewers must correct AI
  output — a direct measure of hallucination impact (AHRQ 2023).

Sources:
  clinical_decisions (75 rows)  — ai_confidence, neurologist_agreement, final_decision
  analyses          (133 rows)  — confidence, predicted_label, signal_quality
  expert_reviews    (3 rows)    — agree_with_ai, role, expert
  hitl_reviews      (4 rows)    — decision (override/accept)
  validation_studies (42 rows)  — sensitivity, specificity, auc_roc
"""

import json
import pathlib
import sqlite3
from collections import Counter, defaultdict

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"

CONFIDENCE_THRESHOLD = 0.80   # high-confidence gate
GROUNDING_GATE_MIN   = 0.75   # minimum grounding score to pass


def _conn():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _grounding_records(cur):
    """Return all clinical_decisions with grounding classification."""
    cur.execute("""
        SELECT id, patient_id, ai_prediction, ai_confidence,
               neurologist_agreement, final_decision, artifact_risk,
               top_channels, note, created_at
        FROM clinical_decisions
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    out = []
    for r in rows:
        conf = float(r["ai_confidence"] or 0)
        agree = (r["neurologist_agreement"] or "").strip().lower()

        # grounding classification
        if agree == "agree":
            grounded = True
            severity = "ok"
        elif agree == "partial":
            grounded = False
            severity = "warning"
        else:  # disagree or unknown
            grounded = False
            severity = "critical" if conf >= CONFIDENCE_THRESHOLD else "warning"

        # hallucination = high confidence + expert disagrees
        hallucination = (conf >= CONFIDENCE_THRESHOLD) and (agree in ("disagree", "partial"))

        out.append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "ai_prediction": r["ai_prediction"],
            "ai_confidence": conf,
            "neurologist_agreement": r["neurologist_agreement"],
            "final_decision": r["final_decision"],
            "artifact_risk": r["artifact_risk"],
            "top_channels": r["top_channels"],
            "note": r["note"],
            "created_at": r["created_at"],
            "grounded": grounded,
            "hallucination": hallucination,
            "severity": severity,
        })
    return out


def _calibration_error(records):
    """Expected Calibration Error (ECE) — binned confidence vs accuracy."""
    bins = defaultdict(lambda: {"n": 0, "correct": 0, "conf_sum": 0})
    for r in records:
        conf = r["ai_confidence"]
        b = round(conf * 10) / 10   # 0.0 … 1.0 in 0.1 steps
        agree = (r["neurologist_agreement"] or "").lower()
        correct = 1 if agree == "agree" else 0
        bins[b]["n"] += 1
        bins[b]["correct"] += correct
        bins[b]["conf_sum"] += conf

    ece = 0.0
    n_total = len(records) or 1
    calib_curve = []
    for b in sorted(bins):
        bn = bins[b]["n"]
        acc = bins[b]["correct"] / bn
        avg_conf = bins[b]["conf_sum"] / bn
        ece += (bn / n_total) * abs(avg_conf - acc)
        calib_curve.append({
            "bin": round(b, 1),
            "accuracy": round(acc, 3),
            "avg_confidence": round(avg_conf, 3),
            "count": bn,
        })
    return round(ece, 4), calib_curve


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def overview():
    con = _conn()
    cur = con.cursor()

    records = _grounding_records(cur)
    n_total = len(records) or 1

    # KPIs
    n_grounded = sum(1 for r in records if r["grounded"])
    n_hallucination = sum(1 for r in records if r["hallucination"])
    n_high_conf = sum(1 for r in records if r["ai_confidence"] >= CONFIDENCE_THRESHOLD)
    avg_conf = sum(r["ai_confidence"] for r in records) / n_total
    grounding_score = round(n_grounded / n_total * 100, 1)
    hallucination_rate = round(n_hallucination / n_total * 100, 1)
    high_conf_accuracy = 0.0
    if n_high_conf:
        high_conf_grounded = sum(1 for r in records if r["ai_confidence"] >= CONFIDENCE_THRESHOLD and r["grounded"])
        high_conf_accuracy = round(high_conf_grounded / n_high_conf * 100, 1)

    ece, calib_curve = _calibration_error(records)

    # HITL override rate
    n_hitl = _safe(cur, "SELECT COUNT(*) FROM hitl_reviews")
    n_override = 0
    for row in cur.execute("SELECT fields_json FROM hitl_reviews"):
        try:
            fj = json.loads(row["fields_json"] or "{}")
            if fj.get("decision") == "override":
                n_override += 1
        except Exception:
            pass
    hitl_override_rate = round(n_override / n_hitl * 100, 1) if n_hitl else 0

    # Expert review agreement
    n_expert = _safe(cur, "SELECT COUNT(*) FROM expert_reviews")
    n_expert_agree = _safe(cur, "SELECT COUNT(*) FROM expert_reviews WHERE agree_with_ai = 'agree'")
    expert_agreement_rate = round(n_expert_agree / n_expert * 100, 1) if n_expert else 0

    # Agreement distribution
    agreement_dist = []
    for row in cur.execute(
        "SELECT neurologist_agreement label, COUNT(*) value FROM clinical_decisions GROUP BY neurologist_agreement"
    ):
        agreement_dist.append({"label": row["label"] or "Unknown", "value": row["value"]})

    # Severity distribution
    sev_counts = Counter(r["severity"] for r in records)
    severity_dist = [
        {"label": "OK (Grounded)", "value": sev_counts.get("ok", 0), "color": "success"},
        {"label": "Warning (Partial)", "value": sev_counts.get("warning", 0), "color": "warning"},
        {"label": "Critical (Hallucination Risk)", "value": sev_counts.get("critical", 0), "color": "danger"},
    ]

    # Confidence distribution histogram
    bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 0.0-0.1 … 0.9-1.0
    for r in records:
        idx = min(int(r["ai_confidence"] * 10), 9)
        bins[idx] += 1
    conf_histogram = [
        {"range": f"{i/10:.1f}–{(i+1)/10:.1f}", "count": bins[i]}
        for i in range(10)
    ]

    # Validation study grounding (AUC, sensitivity, specificity)
    cur.execute("SELECT AVG(auc_roc), AVG(sensitivity), AVG(specificity) FROM validation_studies WHERE auc_roc IS NOT NULL")
    vsrow = cur.fetchone()
    avg_auc = round(vsrow[0], 3) if vsrow and vsrow[0] else None
    avg_sens = round(vsrow[1], 3) if vsrow and vsrow[1] else None
    avg_spec = round(vsrow[2], 3) if vsrow and vsrow[2] else None

    # Gate status
    gate_pass = grounding_score >= (GROUNDING_GATE_MIN * 100)

    con.close()
    return {
        "available": True,
        "gate_status": "PASS" if gate_pass else "FAIL",
        "gate_threshold_pct": GROUNDING_GATE_MIN * 100,
        "confidence_gate": CONFIDENCE_THRESHOLD,
        "kpis": {
            "grounding_score_pct": grounding_score,
            "hallucination_rate_pct": hallucination_rate,
            "high_conf_accuracy_pct": high_conf_accuracy,
            "calibration_error_ece": ece,
            "hitl_override_rate_pct": hitl_override_rate,
            "expert_agreement_rate_pct": expert_agreement_rate,
            "avg_confidence": round(avg_conf, 3),
            "total_decisions": n_total,
            "total_hallucinations": n_hallucination,
            "total_grounded": n_grounded,
            "total_high_conf": n_high_conf,
            "avg_auc": avg_auc,
            "avg_sensitivity": avg_sens,
            "avg_specificity": avg_spec,
        },
        "agreement_distribution": agreement_dist,
        "severity_distribution": severity_dist,
        "confidence_histogram": conf_histogram,
        "calibration_curve": calib_curve,
        "summary": (
            f"Grounding gate {'PASSES' if gate_pass else 'FAILS'} at {grounding_score}% "
            f"(threshold {GROUNDING_GATE_MIN*100}%). "
            f"{n_hallucination} hallucination-risk cases detected "
            f"(high-conf + expert disagreement). ECE={ece:.4f}."
        ),
    }


def breakdown():
    con = _conn()
    cur = con.cursor()

    records = _grounding_records(cur)

    # Hallucination cases — full detail
    hallucination_cases = [
        {k: v for k, v in r.items() if k not in ("grounded", "severity")}
        for r in records if r["hallucination"]
    ]

    # Grounding by confidence band
    bands = [
        ("Low (<0.6)", lambda c: c < 0.6),
        ("Medium (0.6–0.79)", lambda c: 0.6 <= c < 0.80),
        ("High (≥0.80)", lambda c: c >= 0.80),
    ]
    grounding_by_band = []
    for label, fn in bands:
        subset = [r for r in records if fn(r["ai_confidence"])]
        n = len(subset) or 1
        grnd = sum(1 for r in subset if r["grounded"])
        grounding_by_band.append({
            "band": label,
            "total": len(subset),
            "grounded": grnd,
            "grounding_pct": round(grnd / n * 100, 1),
            "hallucinations": sum(1 for r in subset if r["hallucination"]),
        })

    # HITL reviews detail
    hitl_details = []
    for row in cur.execute("SELECT * FROM hitl_reviews ORDER BY created_at DESC"):
        fj = {}
        try:
            fj = json.loads(row["fields_json"] or "{}")
        except Exception:
            pass
        hitl_details.append({
            "id": row["id"],
            "patient_id": row["patient_id"],
            "analysis_id": row["analysis_id"],
            "created_at": row["created_at"],
            **fj,
        })

    # Expert reviews detail
    expert_details = []
    for row in cur.execute("SELECT * FROM expert_reviews ORDER BY created_at DESC"):
        expert_details.append(dict(row))

    # Grounding by patient (top 10 patients by hallucination count)
    patient_counts = defaultdict(lambda: {"decisions": 0, "hallucinations": 0, "grounded": 0})
    for r in records:
        pid = r["patient_id"]
        patient_counts[pid]["decisions"] += 1
        if r["hallucination"]:
            patient_counts[pid]["hallucinations"] += 1
        if r["grounded"]:
            patient_counts[pid]["grounded"] += 1
    patient_summary = sorted(
        [{"patient_id": k, **v} for k, v in patient_counts.items()],
        key=lambda x: -x["hallucinations"],
    )[:15]

    # Calibration curve
    _, calib_curve = _calibration_error(records)

    # Recent decisions (last 20)
    recent_decisions = [
        {k: v for k, v in r.items()}
        for r in records[:20]
    ]

    con.close()
    return {
        "available": True,
        "hallucination_cases": hallucination_cases,
        "grounding_by_confidence_band": grounding_by_band,
        "hitl_reviews": hitl_details,
        "expert_reviews": expert_details,
        "patient_grounding_summary": patient_summary,
        "calibration_curve": calib_curve,
        "recent_decisions": recent_decisions,
    }


def definitions():
    return {
        "available": True,
        "title": "Agent Grounding Gate — Definitions & Methodology",
        "purpose": (
            "The Grounding Gate is a mandatory quality checkpoint that verifies AI "
            "predictions are anchored to verifiable clinical evidence before they "
            "reach the clinician. It detects 'hallucinations' — high-confidence "
            "predictions that cannot be substantiated by expert review."
        ),
        "key_terms": [
            {
                "term": "Grounding Score",
                "definition": "Percentage of AI predictions confirmed by neurologist agreement. "
                              "Target ≥75%. Below threshold triggers gate FAIL.",
                "formula": "Grounded decisions / Total decisions × 100",
            },
            {
                "term": "Hallucination",
                "definition": "An AI prediction with confidence ≥0.80 that a domain expert "
                              "partially or fully disagrees with. Named after the LLM phenomenon "
                              "of confident but unsupported output.",
                "formula": "confidence ≥ 0.80 AND neurologist_agreement ∈ {Disagree, Partial}",
            },
            {
                "term": "Calibration Error (ECE)",
                "definition": "Expected Calibration Error — measures mismatch between stated "
                              "confidence and actual accuracy. ECE=0 is perfect; ECE>0.1 is poor.",
                "formula": "Σ (|bin_accuracy − bin_confidence| × bin_weight)",
            },
            {
                "term": "HITL Override Rate",
                "definition": "Percentage of Human-in-the-Loop reviews that override the AI. "
                              "High override rate signals systematic grounding failure.",
                "formula": "Override decisions / Total HITL reviews × 100",
            },
            {
                "term": "High-Confidence Accuracy",
                "definition": "Among predictions with confidence ≥0.80, the fraction that "
                              "neurologists agree with. Should exceed overall grounding score.",
                "formula": "Grounded(conf≥0.80) / Total(conf≥0.80) × 100",
            },
            {
                "term": "Citation Evidence",
                "definition": "Top channels cited by the AI as evidence. Grounded predictions "
                              "cite specific electrode regions consistent with the seizure focus.",
            },
        ],
        "grounding_levels": [
            {"level": "PASS", "color": "success", "threshold": "Grounding ≥ 75%", "action": "Allow AI output to clinician"},
            {"level": "WARNING", "color": "warning", "threshold": "Grounding 60–74%", "action": "Flag for additional review"},
            {"level": "FAIL", "color": "danger", "threshold": "Grounding < 60%", "action": "Block output; escalate to council"},
        ],
        "severity_classes": [
            {"class": "OK", "description": "Neurologist agrees with AI prediction"},
            {"class": "Warning", "description": "Partial agreement or medium confidence disagreement"},
            {"class": "Critical", "description": "High confidence prediction with expert disagreement — hallucination risk"},
        ],
        "data_sources": [
            {"source": "clinical_decisions", "rows": 75, "fields": "ai_confidence, neurologist_agreement, final_decision"},
            {"source": "expert_reviews", "rows": 3, "fields": "agree_with_ai, role, expert"},
            {"source": "hitl_reviews", "rows": 4, "fields": "decision (override/accept)"},
            {"source": "validation_studies", "rows": 42, "fields": "auc_roc, sensitivity, specificity"},
        ],
        "references": [
            "Rajpurkar et al. (2022) AI in Healthcare — Nature Medicine",
            "Guo et al. (2017) On Calibration of Modern Neural Networks — ICML",
            "AHRQ (2023) AI Safety in Clinical Decision Support",
            "ILAE (2021) Clinical Guidelines for AI-Assisted Epilepsy Diagnosis",
        ],
        "standards_alignment": [
            "IEC 62304 — Software lifecycle processes (AI grounding as mandatory QA gate)",
            "FDA 21 CFR Part 820 — Quality system regulation for AI/ML medical devices",
            "WHO AI Ethics (2021) — Transparency and explainability requirements",
            "ICMR AI Ethics Guidelines (2021) — Human oversight for high-stakes AI",
        ],
    }
