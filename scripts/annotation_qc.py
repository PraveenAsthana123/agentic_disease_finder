#!/usr/bin/env python3
"""Clinical Data Manager — Annotation QC.

Real inter-rater reliability and annotation-quality checks using live data:

1. expert_agreement(): Cohen's κ (2 raters) or Fleiss' κ (≥3) from
   expert_reviews — multiple experts reviewing the same patient/analysis.
2. ai_human_agreement(): agreement rate + κ between AI predictions and
   human HITL override decisions (hitl_reviews).
3. annotation_coverage(): completeness of event_annotations and
   artifact_annotations per patient.
4. annotation_flags(): quality issues — unreviewed patients, disagreements,
   missing annotations on patients with seizure findings.

Reads live tables — report only, no mutation.
"""

import os
import sqlite3
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "clinical.db")


def _cohens_kappa(ratings1, ratings2):
    """Cohen's κ for two raters on the same set of items."""
    if len(ratings1) != len(ratings2) or not ratings1:
        return None
    n = len(ratings1)
    cats = sorted(set(ratings1) | set(ratings2))
    if len(cats) < 2:
        return 1.0  # perfect agreement if only one category used by both

    # Observed agreement
    po = sum(1 for a, b in zip(ratings1, ratings2) if a == b) / n

    # Expected agreement by chance
    cnt1 = Counter(ratings1)
    cnt2 = Counter(ratings2)
    pe = sum((cnt1.get(c, 0) / n) * (cnt2.get(c, 0) / n) for c in cats)

    if pe == 1.0:
        return 1.0
    return round((po - pe) / (1.0 - pe), 4)


def _fleiss_kappa(matrix):
    """Fleiss' κ for ≥3 raters.  matrix[i][j] = number of raters who assigned
    item i to category j."""
    if not matrix:
        return None
    n_items = len(matrix)
    n_raters = sum(matrix[0]) if matrix[0] else 0
    if n_raters < 2:
        return None
    n_cats = len(matrix[0])

    # per-item agreement
    pi = []
    for row in matrix:
        s = sum(row)
        if s < 2:
            continue
        pi.append((sum(r * r for r in row) - s) / (s * (s - 1)))
    if not pi:
        return None
    p_bar = sum(pi) / len(pi)

    # marginal category proportions
    total = n_items * n_raters
    pj = [sum(matrix[i][j] for i in range(n_items)) / total for j in range(n_cats)]
    pe = sum(p * p for p in pj)

    if pe == 1.0:
        return 1.0
    return round((p_bar - pe) / (1.0 - pe), 4)


def _kappa_interpretation(k):
    """Landis & Koch 1977 interpretation."""
    if k is None:
        return "insufficient data"
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def expert_agreement():
    """Inter-rater reliability from expert_reviews."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT patient_id, analysis_id, role, expert, agree_with_ai, finding "
        "FROM expert_reviews ORDER BY patient_id, role")]
    conn.close()

    if not rows:
        return {"available": False, "reason": "no expert_reviews records"}

    # Group by patient — each review is a rater's judgment
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    # Patients with ≥2 reviews → inter-rater comparison
    multi_reviewed = {pid: revs for pid, revs in by_patient.items() if len(revs) >= 2}

    if not multi_reviewed:
        return {
            "available": True,
            "n_reviews": len(rows),
            "n_patients_reviewed": len(by_patient),
            "multi_rater_patients": 0,
            "kappa": None,
            "interpretation": "insufficient data — need ≥2 reviews per patient for κ",
            "note": "Only single-reviewer patients found. κ requires ≥2 raters per item.",
        }

    # Build agreement categories: agree/disagree with AI
    # For Cohen's κ (2 raters): pair first two raters per patient
    # For Fleiss' κ (≥3 raters): build matrix
    categories = ["agree", "disagree"]
    cat_idx = {c: i for i, c in enumerate(categories)}

    n_raters_max = max(len(revs) for revs in multi_reviewed.values())

    if n_raters_max == 2:
        # Cohen's κ between first two raters
        r1_ratings, r2_ratings = [], []
        for pid, revs in multi_reviewed.items():
            r1_ratings.append(revs[0]["agree_with_ai"] or "unknown")
            r2_ratings.append(revs[1]["agree_with_ai"] or "unknown")
        kappa = _cohens_kappa(r1_ratings, r2_ratings)
        method = "Cohen's κ (2 raters)"
    else:
        # Fleiss' κ
        matrix = []
        for pid, revs in multi_reviewed.items():
            row = [0] * len(categories)
            for r in revs:
                verdict = r["agree_with_ai"] or "unknown"
                idx = cat_idx.get(verdict)
                if idx is not None:
                    row[idx] += 1
            matrix.append(row)
        kappa = _fleiss_kappa(matrix)
        method = f"Fleiss' κ ({n_raters_max} raters)"

    # Disagreement details
    disagreements = []
    for pid, revs in multi_reviewed.items():
        verdicts = set(r["agree_with_ai"] for r in revs)
        if len(verdicts) > 1:
            disagreements.append({
                "patient_id": pid,
                "raters": [{"role": r["role"], "expert": r["expert"],
                            "verdict": r["agree_with_ai"],
                            "finding_excerpt": (r["finding"] or "")[:120]}
                           for r in revs],
            })

    return {
        "available": True,
        "method": method,
        "n_reviews": len(rows),
        "n_patients_reviewed": len(by_patient),
        "multi_rater_patients": len(multi_reviewed),
        "kappa": kappa,
        "interpretation": _kappa_interpretation(kappa),
        "agreement_rate": round(1 - len(disagreements) / max(len(multi_reviewed), 1), 4),
        "disagreements": disagreements,
        "note": "Inter-rater agreement on AI classification from live expert_reviews table.",
    }


def ai_human_agreement():
    """Agreement between AI predictions (analyses) and human HITL overrides."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    hitl = [dict(r) for r in conn.execute("SELECT * FROM hitl_reviews")]

    if not hitl:
        conn.close()
        return {"available": False, "reason": "no hitl_reviews records"}

    import json
    comparisons = []
    for r in hitl:
        try:
            fields = json.loads(r["fields_json"]) if r["fields_json"] else {}
        except (json.JSONDecodeError, TypeError):
            fields = {}
        ai_pred = fields.get("ai_prediction", "unknown")
        decision = fields.get("decision", "unknown")  # accept / override
        human_label = fields.get("human_decision", ai_pred) if decision == "override" else ai_pred
        comparisons.append({
            "patient_id": r["patient_id"],
            "ai_prediction": ai_pred,
            "human_decision": human_label,
            "decision_type": decision,
            "reason": fields.get("reason_code", fields.get("reason", "")),
        })

    # Compute agreement
    ai_labels = [c["ai_prediction"] for c in comparisons]
    human_labels = [c["human_decision"] for c in comparisons]
    agreement_count = sum(1 for a, h in zip(ai_labels, human_labels) if a == h)
    override_count = sum(1 for c in comparisons if c["decision_type"] == "override")

    kappa = _cohens_kappa(ai_labels, human_labels)

    conn.close()
    return {
        "available": True,
        "n_reviews": len(comparisons),
        "agreement_count": agreement_count,
        "override_count": override_count,
        "agreement_rate": round(agreement_count / len(comparisons), 4) if comparisons else None,
        "kappa": kappa,
        "interpretation": _kappa_interpretation(kappa),
        "overrides": [c for c in comparisons if c["decision_type"] == "override"],
        "note": "AI-vs-human agreement from hitl_reviews. Overrides flagged for review.",
    }


def annotation_coverage():
    """Completeness of event_annotations and artifact_annotations per patient."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    patients = [dict(r) for r in conn.execute("SELECT patient_id, disease FROM patients")]
    event_annots = conn.execute("SELECT patient_id FROM event_annotations").fetchall()
    artifact_annots = conn.execute("SELECT patient_id FROM artifact_annotations").fetchall()
    eeg_interps = conn.execute("SELECT patient_id FROM eeg_interpretation").fetchall()

    event_pids = set(r[0] for r in event_annots)
    artifact_pids = set(r[0] for r in artifact_annots)
    interp_pids = set(r[0] for r in eeg_interps)
    all_pids = set(p["patient_id"] for p in patients)

    conn.close()

    missing_events = sorted(all_pids - event_pids)
    missing_artifacts = sorted(all_pids - artifact_pids)
    missing_interp = sorted(all_pids - interp_pids)

    return {
        "available": True,
        "n_patients": len(patients),
        "event_annotations": {
            "annotated": len(event_pids),
            "missing": len(missing_events),
            "coverage_pct": round(100 * len(event_pids) / max(len(all_pids), 1), 1),
            "missing_patients": missing_events[:20],
        },
        "artifact_annotations": {
            "annotated": len(artifact_pids),
            "missing": len(missing_artifacts),
            "coverage_pct": round(100 * len(artifact_pids) / max(len(all_pids), 1), 1),
            "missing_patients": missing_artifacts[:20],
        },
        "eeg_interpretation": {
            "annotated": len(interp_pids),
            "missing": len(missing_interp),
            "coverage_pct": round(100 * len(interp_pids) / max(len(all_pids), 1), 1),
            "missing_patients": missing_interp[:20],
        },
        "note": "Annotation completeness per patient across 3 annotation tables.",
    }


def annotation_flags():
    """Quality flags: unreviewed high-risk patients, stale annotations, coverage gaps."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Patients with seizure findings but no expert review
    analyses = [dict(r) for r in conn.execute(
        "SELECT a.patient_id, a.predicted_label, a.confidence "
        "FROM analyses a")]
    reviewed_pids = set(r[0] for r in conn.execute("SELECT DISTINCT patient_id FROM expert_reviews"))
    hitl_pids = set(r[0] for r in conn.execute("SELECT DISTINCT patient_id FROM hitl_reviews"))

    conn.close()

    # Patients with analyses but no expert review
    analyzed_pids = set(a["patient_id"] for a in analyses)
    unreviewed = sorted(analyzed_pids - reviewed_pids - hitl_pids)

    # Low-confidence predictions without human review
    low_conf_unreviewed = [
        {"patient_id": a["patient_id"], "label": a["predicted_label"],
         "confidence": a["confidence"]}
        for a in analyses
        if (a["confidence"] or 0) < 0.65
        and a["patient_id"] not in reviewed_pids
        and a["patient_id"] not in hitl_pids
    ]

    flags = []
    if unreviewed:
        flags.append(f"{len(unreviewed)} patient(s) with AI analysis but no expert/HITL review")
    if low_conf_unreviewed:
        flags.append(f"{len(low_conf_unreviewed)} low-confidence (<0.65) prediction(s) without human review")

    cov = annotation_coverage()
    if cov["event_annotations"]["coverage_pct"] < 50:
        flags.append(f"event_annotations coverage {cov['event_annotations']['coverage_pct']}% — below 50% threshold")
    if cov["artifact_annotations"]["coverage_pct"] < 50:
        flags.append(f"artifact_annotations coverage {cov['artifact_annotations']['coverage_pct']}% — below 50% threshold")

    return {
        "available": True,
        "n_flags": len(flags),
        "flags": flags,
        "unreviewed_patients": unreviewed[:20],
        "low_confidence_unreviewed": low_conf_unreviewed[:10],
        "note": "QC flags surfacing annotation gaps — unreviewed patients, low-confidence without HITL, coverage below threshold.",
    }


def full_report():
    """Complete Annotation QC report."""
    return {
        "role": "Clinical Data Manager — Annotation QC",
        "expert_agreement": expert_agreement(),
        "ai_human_agreement": ai_human_agreement(),
        "annotation_coverage": annotation_coverage(),
        "annotation_flags": annotation_flags(),
    }


if __name__ == "__main__":
    import json
    r = full_report()
    print(json.dumps(r, indent=2, default=str))
